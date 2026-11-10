// SPDX-License-Identifier: Apache-2.0
#pragma once
inline constexpr wchar_t theme_xaml[] = LR"xaml(
<ResourceDictionary xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
 <ResourceDictionary.ThemeDictionaries>
  <ResourceDictionary x:Key="Default">
   <SolidColorBrush x:Key="ProgressBarForeground" Color="#76B900"/>
   <SolidColorBrush x:Key="InfoBadgeBackground" Color="#76B900"/>
   <SolidColorBrush x:Key="InfoBadgeForeground" Color="#101700"/>
   <SolidColorBrush x:Key="SliderThumbBackground" Color="#76B900"/>
   <SolidColorBrush x:Key="SliderThumbBackgroundPointerOver" Color="#8ACC18"/>
   <SolidColorBrush x:Key="SliderThumbBackgroundPressed" Color="#679F00"/>
   <SolidColorBrush x:Key="SliderTrackValueFill" Color="#76B900"/>
   <SolidColorBrush x:Key="SliderTrackValueFillPointerOver" Color="#8ACC18"/>
   <SolidColorBrush x:Key="SliderTrackValueFillPressed" Color="#679F00"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOn" Color="#76B900"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOnPointerOver" Color="#8ACC18"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOnPressed" Color="#679F00"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOn" Color="#76B900"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOnPointerOver" Color="#8ACC18"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOnPressed" Color="#679F00"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOn" Color="#101010"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOnPointerOver" Color="#101010"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOnPressed" Color="#101010"/>
   <SolidColorBrush x:Key="ShellBrush">#0B0B0B</SolidColorBrush><SolidColorBrush x:Key="WorkspaceBrush">#111111</SolidColorBrush><SolidColorBrush x:Key="SurfaceBrush">#1B1B1B</SolidColorBrush><SolidColorBrush x:Key="StrokeBrush">#333333</SolidColorBrush>
   <SolidColorBrush x:Key="AppTextBrush">#F5F5F5</SolidColorBrush><SolidColorBrush x:Key="SecondaryTextBrush">#B4B4B4</SolidColorBrush><SolidColorBrush x:Key="AppAccentBrush">#76B900</SolidColorBrush><SolidColorBrush x:Key="SelectedBrush">#242424</SolidColorBrush><SolidColorBrush x:Key="PositiveBrush">#A5D956</SolidColorBrush>
   <SolidColorBrush x:Key="PrefillSeriesBrush">#55B9DA</SolidColorBrush><SolidColorBrush x:Key="VerifySeriesBrush">#C2A6EF</SolidColorBrush>
   <CornerRadius x:Key="ControlCornerRadius">3</CornerRadius><CornerRadius x:Key="OverlayCornerRadius">4</CornerRadius>
   <SolidColorBrush x:Key="NavigationViewDefaultPaneBackground">#0B0B0B</SolidColorBrush><SolidColorBrush x:Key="NavigationViewContentBackground">#111111</SolidColorBrush><SolidColorBrush x:Key="NavigationViewItemBackgroundSelected">#242424</SolidColorBrush>
   <SolidColorBrush x:Key="ButtonBackground">#202020</SolidColorBrush><SolidColorBrush x:Key="ButtonBackgroundPointerOver">#2B2B2B</SolidColorBrush><SolidColorBrush x:Key="ButtonBackgroundPressed">#171717</SolidColorBrush><SolidColorBrush x:Key="ButtonBorderBrush">#3A3A3A</SolidColorBrush>
   <SolidColorBrush x:Key="TextControlBackground">#161616</SolidColorBrush><SolidColorBrush x:Key="TextControlBackgroundPointerOver">#202020</SolidColorBrush><SolidColorBrush x:Key="TextControlBackgroundFocused">#0B0B0B</SolidColorBrush>
   <SolidColorBrush x:Key="ContentDialogBackground">#111111</SolidColorBrush><SolidColorBrush x:Key="ExpanderHeaderBackground">#1B1B1B</SolidColorBrush>
   <Color x:Key="SystemAccentColor">#76B900</Color><SolidColorBrush x:Key="SystemControlHighlightAccentBrush">#76B900</SolidColorBrush>
   <SolidColorBrush x:Key="AccentButtonBackground">#76B900</SolidColorBrush><SolidColorBrush x:Key="AccentButtonBackgroundPointerOver">#8ACC18</SolidColorBrush><SolidColorBrush x:Key="AccentButtonBackgroundPressed">#679F00</SolidColorBrush>
   <SolidColorBrush x:Key="AccentButtonForeground">#101700</SolidColorBrush><SolidColorBrush x:Key="AccentButtonForegroundPointerOver">#101700</SolidColorBrush><SolidColorBrush x:Key="AccentButtonForegroundPressed">#101700</SolidColorBrush>
     <SolidColorBrush x:Key="NavigationViewSelectionIndicatorForeground" Color="#76B900"/>
   <SolidColorBrush x:Key="TextControlBorderBrushFocused" Color="#76B900"/>
   <SolidColorBrush x:Key="TextControlElevationBorderFocusedBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="AccentTextFillColorPrimaryBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="AccentTextFillColorSecondaryBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="AccentFillColorDefaultBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="AccentFillColorSecondaryBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="AccentFillColorTertiaryBrush" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillChecked" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillCheckedPointerOver" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillCheckedPressed" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeChecked" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeCheckedPointerOver" Color="#76B900"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeCheckedPressed" Color="#76B900"/>
   <SolidColorBrush x:Key="TextControlSelectionHighlightColor" Color="#497100"/>
   <SolidColorBrush x:Key="TextSelectionHighlightColorThemeBrush" Color="#497100"/>
   <SolidColorBrush x:Key="AccentFillColorSelectedTextBackgroundBrush" Color="#497100"/>
   <SolidColorBrush x:Key="TextOnAccentFillColorPrimaryBrush" Color="#101700"/>
   <SolidColorBrush x:Key="TextOnAccentFillColorSecondaryBrush" Color="#101700"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundChecked" Color="#101700"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundCheckedPointerOver" Color="#101700"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundCheckedPressed" Color="#101700"/>
  </ResourceDictionary>
)xaml"
LR"xaml(

  <ResourceDictionary x:Key="HighContrast">
   <SolidColorBrush x:Key="ProgressBarForeground" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="InfoBadgeBackground" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="InfoBadgeForeground" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="SliderThumbBackground" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="SliderThumbBackgroundPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="SliderThumbBackgroundPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="SliderTrackValueFill" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="SliderTrackValueFillPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="SliderTrackValueFillPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOn" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOnPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchFillOnPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOn" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOnPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchStrokeOnPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOn" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOnPointerOver" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="ToggleSwitchKnobFillOnPressed" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="PrefillSeriesBrush" Color="{ThemeResource SystemColorWindowTextColor}"/><SolidColorBrush x:Key="VerifySeriesBrush" Color="{ThemeResource SystemColorWindowTextColor}"/>
   <SolidColorBrush x:Key="NavigationViewDefaultPaneBackground" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="NavigationViewContentBackground" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="NavigationViewItemBackgroundSelected" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="ButtonBackground" Color="{ThemeResource SystemColorButtonFaceColor}"/><SolidColorBrush x:Key="ButtonBackgroundPointerOver" Color="{ThemeResource SystemColorHighlightTextColor}"/><SolidColorBrush x:Key="ButtonBackgroundPressed" Color="{ThemeResource SystemColorHighlightTextColor}"/><SolidColorBrush x:Key="ButtonBorderBrush" Color="{ThemeResource SystemColorWindowTextColor}"/>
   <SolidColorBrush x:Key="TextControlBackground" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="TextControlBackgroundPointerOver" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="TextControlBackgroundFocused" Color="{ThemeResource SystemColorWindowColor}"/>
   <SolidColorBrush x:Key="ContentDialogBackground" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="ExpanderHeaderBackground" Color="{ThemeResource SystemColorButtonFaceColor}"/>
   <SolidColorBrush x:Key="ShellBrush" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="WorkspaceBrush" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="SurfaceBrush" Color="{ThemeResource SystemColorWindowColor}"/><SolidColorBrush x:Key="StrokeBrush" Color="{ThemeResource SystemColorWindowTextColor}"/>
   <SolidColorBrush x:Key="AppTextBrush" Color="{ThemeResource SystemColorWindowTextColor}"/><SolidColorBrush x:Key="SecondaryTextBrush" Color="{ThemeResource SystemColorWindowTextColor}"/><SolidColorBrush x:Key="AppAccentBrush" Color="{ThemeResource SystemColorHighlightColor}"/><SolidColorBrush x:Key="SelectedBrush" Color="{ThemeResource SystemColorHighlightColor}"/><SolidColorBrush x:Key="PositiveBrush" Color="{ThemeResource SystemColorWindowTextColor}"/>
   <SolidColorBrush x:Key="AccentButtonBackground" Color="{ThemeResource SystemColorHighlightColor}"/><SolidColorBrush x:Key="AccentButtonBackgroundPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/><SolidColorBrush x:Key="AccentButtonBackgroundPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentButtonForeground" Color="{ThemeResource SystemColorHighlightTextColor}"/><SolidColorBrush x:Key="AccentButtonForegroundPointerOver" Color="{ThemeResource SystemColorHighlightTextColor}"/><SolidColorBrush x:Key="AccentButtonForegroundPressed" Color="{ThemeResource SystemColorHighlightTextColor}"/>
     <SolidColorBrush x:Key="NavigationViewSelectionIndicatorForeground" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="TextControlBorderBrushFocused" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="TextControlElevationBorderFocusedBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
)xaml"
LR"xaml(
   <SolidColorBrush x:Key="AccentTextFillColorPrimaryBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentTextFillColorSecondaryBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentFillColorDefaultBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentFillColorSecondaryBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentFillColorTertiaryBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillChecked" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillCheckedPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundFillCheckedPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeChecked" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeCheckedPointerOver" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckBackgroundStrokeCheckedPressed" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="TextControlSelectionHighlightColor" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="TextSelectionHighlightColorThemeBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="AccentFillColorSelectedTextBackgroundBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
   <SolidColorBrush x:Key="TextOnAccentFillColorPrimaryBrush" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="TextOnAccentFillColorSecondaryBrush" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundChecked" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundCheckedPointerOver" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="CheckBoxCheckGlyphForegroundCheckedPressed" Color="{ThemeResource SystemColorHighlightTextColor}"/>
   <SolidColorBrush x:Key="SystemControlHighlightAccentBrush" Color="{ThemeResource SystemColorHighlightColor}"/>
  </ResourceDictionary>
 </ResourceDictionary.ThemeDictionaries>
 <Style TargetType="TextBlock"><Setter Property="Foreground" Value="{ThemeResource AppTextBrush}"/><Setter Property="FontFamily" Value="Segoe UI Variable"/><Setter Property="FontSize" Value="14"/></Style>
 <Style x:Key="Secondary" TargetType="TextBlock"><Setter Property="Foreground" Value="{ThemeResource SecondaryTextBrush}"/><Setter Property="TextWrapping" Value="Wrap"/></Style>
 <Style x:Key="Caption" TargetType="TextBlock" BasedOn="{StaticResource Secondary}"><Setter Property="FontSize" Value="12"/></Style>
 <Style x:Key="Card" TargetType="Border"><Setter Property="Background" Value="{ThemeResource SurfaceBrush}"/><Setter Property="Padding" Value="16"/><Setter Property="CornerRadius" Value="4"/><Setter Property="BorderBrush" Value="{ThemeResource StrokeBrush}"/><Setter Property="BorderThickness" Value="1"/></Style>
 <Style x:Key="TextAction" TargetType="Button"><Setter Property="Background" Value="Transparent"/><Setter Property="BorderThickness" Value="0"/><Setter Property="Padding" Value="0,7"/><Setter Property="HorizontalAlignment" Value="Left"/></Style>
</ResourceDictionary>
)xaml";
