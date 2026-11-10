// SPDX-License-Identifier: Apache-2.0
    struct ChatRow {
        Border root{nullptr};
        TextBlock content{nullptr},reasoning{nullptr},status{nullptr};
        Expander reasoning_panel{nullptr};
        Button copy{nullptr};
        std::shared_ptr<std::wstring> copy_text;
        std::wstring role,last_content,last_reasoning,last_status;
    };
    void initialize_chat() {
        clicked(L"ChatSendButton",[this]{send_chat();});
        clicked(L"ChatStopButton",[this]{if(!flare_app_cancel_preview(app_))chat_error_=wide(flare_last_error());else chat_stop_pending_=true;update_chat_controls();});
        clicked(L"ChatNewButton",[this]{new_chat();});
        for(auto name:{L"ChatChooseButton",L"ChatEmptyChooseButton"})clicked(name,[this]{choose_model();});
        clicked(L"ChatExplainSuggestion",[this]{control<TextBox>(L"ChatComposer").Text(L"Explain a topic to me in simple terms: ");control<TextBox>(L"ChatComposer").Focus(FocusState::Programmatic);});
        clicked(L"ChatIdeasSuggestion",[this]{control<TextBox>(L"ChatComposer").Text(L"Help me think through an idea: ");control<TextBox>(L"ChatComposer").Focus(FocusState::Programmatic);});
        auto composer=control<TextBox>(L"ChatComposer");
        composer.TextChanged([this](auto const&,auto const&){update_chat_controls();});
        composer.TextCompositionStarted([this](auto const&,auto const&){chat_composing_=true;});
        composer.TextCompositionEnded([this](auto const&,auto const&){chat_composing_=false;});
        composer.PreviewKeyDown([this](auto const&,Microsoft::UI::Xaml::Input::KeyRoutedEventArgs const& args){
            if(args.Key()==Windows::System::VirtualKey::Enter&&!chat_composing_&&!(GetKeyState(VK_SHIFT)&0x8000)){args.Handled(true);if(!args.KeyStatus().WasKeyDown)send_chat();}
        });
        initialize_chat_scroll();
    }
    void initialize_chat_scroll() {
        control<ScrollViewer>(L"ChatScroll").SizeChanged([this](auto const&,auto const&){size_chat_thread();});
        control<ScrollViewer>(L"ChatScroll").ViewChanged([this](auto const&,auto const&){
            size_chat_thread();
            auto scroll=control<ScrollViewer>(L"ChatScroll");auto offset=scroll.VerticalOffset();
            bool moved=std::abs(offset-chat_last_offset_)>0.5;chat_last_offset_=offset;
            if(chat_auto_scrolling_||!moved)return;
            chat_follow_=scroll.ScrollableHeight()-offset<72;
            if(!chat_follow_)chat_scroll_pending_=false;
        });
        control<StackPanel>(L"ChatThread").SizeChanged([this](auto const&,auto const&){if(chat_follow_)chat_scroll_pending_=true;});
        control<ScrollViewer>(L"ChatScroll").LayoutUpdated([this](auto const&,auto const&){
            if(!chat_scroll_pending_||page_!=L"Chat")return;
            chat_scroll_pending_=false;chat_auto_scrolling_=true;
            auto scroll=control<ScrollViewer>(L"ChatScroll");scroll.ChangeView(nullptr,box_value(scroll.ScrollableHeight()).as<IReference<double>>(),nullptr,true);
            chat_auto_scrolling_=false;
        });
    }
    void size_chat_thread() {
        auto scroll=control<ScrollViewer>(L"ChatScroll");auto thread=control<StackPanel>(L"ChatThread");
        auto viewport=scroll.ViewportWidth()>0?scroll.ViewportWidth():scroll.ActualWidth();
        auto width=std::max(1.0,std::min(720.0,viewport-16.0));
        if(!std::isfinite(thread.Width())||std::abs(thread.Width()-width)>0.5)thread.Width(width);
    }
    void send_chat() {
        if(closing_||!chat_ready_||chat_busy_||chat_submit_pending_)return;
        auto composer=control<TextBox>(L"ChatComposer");auto message=std::wstring(composer.Text());
        if(std::all_of(message.begin(),message.end(),[](wchar_t c){return std::iswspace(c)!=0;}))return;
        auto encoded=to_string(message);auto operation=flare_app_try_model(app_,encoded.data(),encoded.size());
        if(!operation){chat_error_=wide(flare_last_error());update_chat_controls();return;}
        chat_operation_=operation;chat_submit_pending_=true;chat_error_.clear();chat_submitted_text_=message;
        composer.Text(L"");update_chat_controls();composer.Focus(FocusState::Programmatic);
    }
    void new_chat() {
        if(chat_busy_||chat_submit_pending_)return;
        if(!flare_app_clear_chat(app_)){chat_error_=wide(flare_last_error());update_chat_controls();return;}
        chat_conversation_=UINT64_MAX;chat_error_.clear();chat_submitted_text_.clear();chat_operation_=0;
        chat_rows_.clear();control<StackPanel>(L"ChatThread").Children().Clear();
        chat_follow_=true;chat_scroll_pending_=true;
        control<TextBox>(L"ChatComposer").Text(L"");update_chat_controls();
        control<TextBox>(L"ChatComposer").Focus(FocusState::Programmatic);
    }
    void update_chat_controls() {
        bool busy=chat_busy_||chat_submit_pending_;
        auto draft=std::wstring(control<TextBox>(L"ChatComposer").Text());
        bool has_text=std::any_of(draft.begin(),draft.end(),[](wchar_t c){return !std::iswspace(c);});
        control<Button>(L"ChatSendButton").IsEnabled(chat_ready_&&!busy&&has_text);
        control<Button>(L"ChatStopButton").Visibility(busy?Visibility::Visible:Visibility::Collapsed);
        control<Button>(L"ChatStopButton").IsEnabled(chat_busy_&&!chat_stop_pending_);
        control<Button>(L"ChatNewButton").IsEnabled(!busy);
        control<Button>(L"ChatChooseButton").IsEnabled(can_change_model_);
        control<Button>(L"ChatEmptyChooseButton").IsEnabled(can_change_model_);
        control<Button>(L"ChatChooseButton").Content(box_value(chat_ready_?L"Switch model":L"Choose a model"));
        control<StackPanel>(L"ChatSuggestions").Visibility(chat_ready_?Visibility::Visible:Visibility::Collapsed);
        control<Button>(L"ChatEmptyChooseButton").Visibility(chat_ready_?Visibility::Collapsed:Visibility::Visible);
        auto model=active_name_.empty()?pending_model_name_:active_name_;
        text(L"ChatModelName",model.empty()?L"No model loaded":model+L" · "+(chat_ready_?L"Ready":state_name(session_state_)));
        ToolTipService::SetToolTip(control<TextBlock>(L"ChatModelName"),box_value(model));
        text(L"ChatEmptyHint",chat_ready_?L"Send a quick message to check its responses, then connect it to your apps from Home.":session_state_==FLARE_LOADING?L"Your model is getting ready. You can write a test message while it loads.":L"Choose a local model to try a few messages before connecting your apps.");
        text(L"ChatStatus",!chat_error_.empty()?chat_error_:chat_submit_pending_?L"Sending to your local model…":busy?(chat_stop_pending_?L"Stopping…":L"Generating on this PC…"):!chat_ready_?L"Load a model to send messages.":L"Ready to chat · Your conversation stays in memory.");
    }
    ChatRow create_chat_row(std::wstring const& role) {
        ChatRow row;row.role=role;row.copy_text=std::make_shared<std::wstring>();
        bool user=role==L"user";
        row.root=Markup::XamlReader::Load(user?
            L"<Border xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation' CornerRadius='8,8,2,8' Padding='16,12' Background='{ThemeResource SurfaceBrush}' BorderBrush='{ThemeResource StrokeBrush}' BorderThickness='1'/>" :
            L"<Border xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation'/>").as<Border>();
        row.root.HorizontalAlignment(user?HorizontalAlignment::Right:HorizontalAlignment::Stretch);
        if(user)row.root.MaxWidth(560);
        StackPanel body;body.Spacing(10);row.root.Child(body);
        auto speaker=label(user?L"You":L"Flare",12,true);speaker.FontWeight(Windows::UI::Text::FontWeights::SemiBold());body.Children().Append(speaker);
        row.reasoning=label(L"",12,true);row.reasoning.LineHeight(20);
        ScrollViewer reasoning_scroll;reasoning_scroll.MaxHeight(240);reasoning_scroll.HorizontalScrollMode(ScrollMode::Disabled);reasoning_scroll.HorizontalScrollBarVisibility(ScrollBarVisibility::Disabled);reasoning_scroll.HorizontalContentAlignment(HorizontalAlignment::Stretch);reasoning_scroll.Content(row.reasoning);
        row.reasoning_panel=Expander();row.reasoning_panel.Header(box_value(L"Reasoning"));row.reasoning_panel.HorizontalAlignment(HorizontalAlignment::Stretch);row.reasoning_panel.Content(reasoning_scroll);row.reasoning_panel.Visibility(Visibility::Collapsed);body.Children().Append(row.reasoning_panel);
        row.content=label(L"",14);row.content.LineHeight(22);body.Children().Append(row.content);
        row.status=label(L"",12,true);row.status.Visibility(Visibility::Collapsed);body.Children().Append(row.status);
        row.copy=Button();row.copy.Style(theme_resources_.Lookup(box_value(L"TextAction")).as<Style>());row.copy.Content(box_value(L"Copy"));row.copy.FontSize(12);row.copy.Visibility(Visibility::Collapsed);
        Automation::AutomationProperties::SetName(row.copy,L"Copy assistant response");
        row.copy.Click([this,value=row.copy_text](IInspectable const& sender,RoutedEventArgs const&){
            try {copy_text(*value);sender.as<Button>().Content(box_value(L"Copied"));}
            catch(hresult_error const& error){show_error(std::wstring(error.message()));}
        });body.Children().Append(row.copy);
        return row;
    }
    void update_chat(flare_snapshot const* snapshot,flare_status const& status) {
        flare_preview value{};if(!flare_snapshot_preview(snapshot,&value))return;
        chat_ready_=status.state==FLARE_READY&&status.accepting;
        bool session_changed=chat_session_!=status.session_id;
        if(session_changed||chat_conversation_!=value.conversation_id){
            chat_rows_.clear();control<StackPanel>(L"ChatThread").Children().Clear();
            chat_follow_=true;chat_scroll_pending_=true;
            if(session_changed){chat_error_.clear();chat_submitted_text_.clear();chat_operation_=0;chat_submit_pending_=false;}
            chat_session_=status.session_id;chat_conversation_=value.conversation_id;
        }
        chat_busy_=value.busy!=0;
        if(value.operation_id==chat_operation_)chat_submit_pending_=false;
        if(!chat_busy_)chat_stop_pending_=false;
        if(chat_operation_&&value.operation_id==chat_operation_&&!chat_busy_&&!chat_submitted_text_.empty()){
            if(value.error.length){chat_error_=wide(value.error);auto composer=control<TextBox>(L"ChatComposer");if(composer.Text().empty())composer.Text(chat_submitted_text_);}
            chat_submitted_text_.clear();
        }
        update_chat_controls();
        if(page_!=L"Chat")return;
        render_chat_messages(snapshot,value.session_id==status.session_id?flare_snapshot_chat_message_count(snapshot):0);
    }
    void render_chat_messages(flare_snapshot const* snapshot,size_t count) {
        auto thread=control<StackPanel>(L"ChatThread");
        if(count<chat_rows_.size()){chat_rows_.clear();thread.Children().Clear();}
        control<ScrollViewer>(L"ChatEmptyHost").Visibility(count?Visibility::Collapsed:Visibility::Visible);
        for(size_t i=0;i<count;++i){
            flare_chat_message message{};if(!flare_snapshot_chat_message(snapshot,i,&message))continue;
            if(i==chat_rows_.size()){auto row=create_chat_row(wide(message.role));thread.Children().Append(row.root);chat_rows_.push_back(std::move(row));}
            update_chat_row(chat_rows_[i],message,chat_busy_&&i+1==count);
        }
    }
    void update_chat_row(ChatRow& row,flare_chat_message const& value,bool generating) {
        auto content=wide(value.content),reasoning=wide(value.reasoning);
        std::wstring status=value.error.length?wide(value.error):value.cancelled?L"Stopped · Partial response":generating?L"Generating…":L"";
        bool changed=false;
        if(content!=row.last_content){row.content.Text(content);row.content.Visibility(content.empty()?Visibility::Collapsed:Visibility::Visible);row.last_content=content;*row.copy_text=content;row.copy.Content(box_value(L"Copy"));changed=true;}
        if(reasoning!=row.last_reasoning){row.reasoning.Text(reasoning);row.last_reasoning=reasoning;changed=true;}
        if(status!=row.last_status){row.status.Text(status);row.status.Visibility(status.empty()?Visibility::Collapsed:Visibility::Visible);row.last_status=status;changed=true;}
        row.reasoning_panel.Visibility(reasoning.empty()?Visibility::Collapsed:Visibility::Visible);
        row.reasoning_panel.Header(box_value(generating?L"Thinking":L"Reasoning"));
        row.copy.Visibility(row.role!=L"user"&&!content.empty()?Visibility::Visible:Visibility::Collapsed);
        if(changed&&chat_follow_)chat_scroll_pending_=true;
    }
